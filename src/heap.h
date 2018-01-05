#ifndef HEAP_H
#define HEAP_H
#include <vector>
#include <utility>
#include <iostream>
using std::cout;
using std::endl;
namespace my{
    inline unsigned int left(unsigned int i) {
        return 2*i+1;
    }
    inline unsigned int right(unsigned int i) {
        return 2*i+2;
    }
    
    template<typename T, typename U>
    void heapify(T *array, unsigned int i, unsigned int n, std::vector<U> &heap) {
        unsigned int lowest;
        lowest = left(i);
        if (right(i) < n) {
            if (array[heap[left(i)]] > array[heap[right(i)]]) {
                lowest = right(i);
            }
        } 
        if (array[heap[lowest]] < array[heap[i]]) {
            std::swap(heap[lowest],heap[i]);
        }
        if (lowest*2+1 < n) {
            heapify(array, lowest, n, heap);
        }
    }
    template<typename T, typename U> 
    void make_heap(T *array, unsigned int n, std::vector<U> &heap ) { 
        unsigned int i;
        for (i=0; i<n; i++) heap.push_back(i);
        for (i=(n-2)/2+1; i>0; i--) {
            heapify(array, i-1, n, heap);
        }
    }
    
    template<typename T, typename U>
    U pop_heap(T *array, unsigned int n, std::vector<U> &heap) {
        unsigned int root = heap[0]; 
        heap[0] = heap.back();
        heap.pop_back();
        heapify(array, 0, n-1, heap);
        return root;
    }
}
#endif
