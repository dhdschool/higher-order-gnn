#include <stdio.h>
#include <stdlib.h>

typedef struct{
    int* arr_ptr;
    int arr_size_0;
} vec;

typedef struct{
    int* arr_ptr;
    int arr_size_0;
    int arr_size_1;
} mat;

long* FACT_TABLE;
long factorial(int n){
    long total=1;
    for(int i=2; i<n; i++){
        total = total * i;
    }
    return total;
}

long* comb(int n, int k){
    return factorial(n) / (factorial(k) * factorial(n-k));
}



mat combinations(vec x, int k){
    int n = x.arr_size_0;
    int m = (int)comb(n, k);

    int* arr_ptr = ;
}

int main(){
    FACT_TABLE = (int*)calloc(20, sizeof(int));
    for(int i=0; i<21; i++){
        FACT_TABLE[i] = factorial(i);
    }

    return 0;
}