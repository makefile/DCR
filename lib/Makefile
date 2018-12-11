OS := linux
SCRIPT := setup_${OS}.py
all:
	cd nms/; python ${SCRIPT} build_ext --inplace; rm -rf build; cd ../../
	cd bbox/; python ${SCRIPT} build_ext --inplace; rm -rf build; cd ../../
	cd dataset/pycocotools/; python ${SCRIPT} build_ext --inplace; rm -rf build; cd ../../
clean:
	cd nms/; rm *.so *.c *.cpp; cd ../../
	cd bbox/; rm *.so *.c *.cpp; cd ../../
	cd dataset/pycocotools/; rm *.so; cd ../../
