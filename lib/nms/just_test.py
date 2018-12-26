
def in_rect( pt_x,  pt_y, pts) :
    # // https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
    # print type(pts)
    isInside = False;
    # // the previous bounding box check can remove false-negatives in edge-cases
    # // remove the previous check code to speed up if you don't care of edge-cases
    n = 4 # point num
    j = n - 1
    for i in range(n):

        ix = pts[i * 2]
        iy = pts[i * 2 + 1];
        jx = pts[j * 2]
        jy = pts[j * 2 + 1];
        if ( (iy > pt_y) != (jy > pt_y) and \
                pt_x < (jx - ix) * (pt_y - iy) / (jy - iy) + ix ):
            isInside = not isInside

        j = i; i+=1

    return isInside

if __name__ == '__main__':
    pts = [10.2,5.2, 55.2,5.2, 55.2,45.2, 10.2,45.2]
    cnt = 0
    for i in range(100):
        for j in range(100):
            if in_rect(i, j, pts):
                cnt += 1
    print cnt
