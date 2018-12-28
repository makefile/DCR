# --------------------------------------------------------
# fetch data in threading
# better way is to use multiprocessing instead of threading, since our data processing is CPU intensive
# however it is easy to cause CUDA initialization error in multiprocessing
# write by fyk
# --------------------------------------------------------

import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice

from config.config import config
from rpn.rpn_rotate import get_rpn_batch_quadrangle, assign_quadrangle_anchor

# import multiprocessing
import threading
import Queue
import atexit

class ThreadedQuadrangleAnchorLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, anchor_angles=(-60, -30, 0, 30, 60, 90), n_thread = 7, inclined_anchor=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(ThreadedQuadrangleAnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_angles = anchor_angles
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.inclined_anchor = inclined_anchor

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        if shuffle: self.shuffle_idx()
        self.data, self.label = self.get_batch_individual(self.cur)

        self.n_thread = n_thread
        # self.result_q = multiprocessing.Queue(maxsize=self.batch_size * 32)
        # self.request_q = multiprocessing.Queue()
        self.result_q = Queue.Queue(maxsize=n_thread)
        self.request_q = Queue.Queue()

        self.stop_flag = False
        self.worker_proc = None
        self.stop_word = '==STOP--'
        # CUDA error: initialization error
        # you need to make sure there are no CUDA calls prior to starting your subprocesses
        # otherwise you can set start method as spawn as a workaround
        # python3: multiprocessing.set_start_method('spawn')
        self.reset_process()

    def _thread_start(self):
        '''
        create process for loading data to blocking queue
        :return:
        '''
        self.stop_flag = False
        # self.worker_proc = [multiprocessing.Process(target=self._worker,
        self.worker_proc = [threading.Thread(target=self._worker,
                                       args=[pid,
                                             self.request_q,
                                             self.result_q])
                            for pid in range(self.n_thread)]
        # [item.start() for item in self.worker_proc]
        for worker in self.worker_proc:
            worker.daemon = True
            worker.start()
        '''
        fyk: we do not need set daemon flag, since we handle it by our self
        The process's daemon flag, a Boolean value. This must be set before start() is called.
        The initial value is inherited from the creating process.
        When a process exits, it attempts to terminate all of its daemonic child processes.
        Note that a daemonic process is not allowed to create child processes. 
        Otherwise a daemonic process would leave its children orphaned if it gets terminated when its parent process exits. 
        Additionally, these are not Unix daemons or services, 
        they are normal processes that will be terminated (and not joined) if non-daemonic processes have exited.
        '''

        def cleanup():
            self.stop_flag = True
            # self.shutdown()
        atexit.register(cleanup)

    def _worker(self, worker_id, data_queue, result_queue):
        # count = 0
        for item in iter(data_queue.get, self.stop_word): # call get until met stop_word
            if self.stop_flag: break
            # print 'worker-{} get data idx-{}'.format(worker_id, item)
            data, label = self.get_batch_individual(item)
            # default param: block=True, allow block when queue is full; timeout=None, never timeout
            result_queue.put((data, label))
            # count += 1

    def shutdown(self):
        # clean queue
        while True:
            try:
                self.result_q.get(timeout=1)
            except Queue.Empty:
                break
        while True:
            try:
                self.request_q.get(timeout=1)
            except Queue.Empty:
                break
        # stop worker
        self.stop_flag = True
        if self.worker_proc:
            for i, worker in enumerate(self.worker_proc):
                worker.join(timeout=1)
                # if worker.is_alive():
                    # logging.error('worker {} is join fail'.format(i))
                    # worker.terminate()

    def __del__(self):
        self.shutdown()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset_process(self):
        self.shutdown()
        [self.request_q.put(i) for i in range(0, len(self.index), self.batch_size)]
        [self.request_q.put(self.stop_word) for pid in range(self.n_thread)]
        self._thread_start()

    def shuffle_idx(self):
        if self.aspect_grouping:
            widths = np.array([r['width'] for r in self.roidb])
            heights = np.array([r['height'] for r in self.roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
            extra = inds.shape[0] % self.batch_size
            inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
            row_perm = np.random.permutation(np.arange(inds_.shape[0]))
            inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
            self.index = inds
        else:
            np.random.shuffle(self.index)

    def reset(self):
        self.cur = 0
        if self.shuffle:
            self.shuffle_idx()

        self.reset_process()

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.data, self.label = self.result_q.get()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        # change the shape of im_info
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_quadrangle_anchor(feat_shape[0], np.zeros((0, 9)), im_info, self.cfg,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios,
                              self.anchor_angles, self.inclined_anchor,
                              self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch_individual(self, cur_from):
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))
        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        return data, label

    def parfetch(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_batch_quadrangle(iroidb, self.cfg)
        # print data
        # print label
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_quadrangle_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.anchor_angles, self.inclined_anchor,
                              self.allowed_border)
        return {'data': data, 'label': label}


