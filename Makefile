spmv_cuda:
	/usr/local/cuda-10.0/bin/nvcc main.cu -o spmv_cuda
clean:
	rm spmv_cuda
