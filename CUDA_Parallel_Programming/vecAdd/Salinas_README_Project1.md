
# Compiling CUDA Programs
- Name: Jose Salinas
- Course: CS-6370
- Professor: Meilin Liu
- Date: 9/15/2026
- Project Name: Project 1 VecAdd VecMult

## Matrix Add Program
- To compile  ```nvcc Salinas_matrixAdd_Project_1.cu -o matrixAdd```
- To start program ```./matrixAdd```
- This will prompt a 2 inputs from the user
```
Enter Thread Block Size: <Thread Block Width>
Enter Matrix Size: <Matrix Width>
```
- After it will output full matrix if matrix size  <= 8, otherwise it will output the first 20 elements. It will then verify that CPU and GPU outputs are the same.

## Matrix Multiply Program
- To compile```nvcc Salinas_matrixMult_Project_1.cu -o matrixMult```
- To start program ```./matrixMult```
- This will prompt a 2 inputs from the user
```
Enter Thread Block Size: <Thread Block Width>
Enter Matrix Size: <Matrix Width>
```
- After it will output full matrix if matrix size  <= 8, otherwise it will output the first 20 elements. It will then verify that CPU and GPU outputs are the same.



