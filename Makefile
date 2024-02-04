tool:
	mkdir -p build && cd build && cmake ../ && cmake --build .

run:
	./build/cuda_ex

vis:
	python visualization.py

clean:
	rm *.png
	rm *.csv
	rm -rf build
