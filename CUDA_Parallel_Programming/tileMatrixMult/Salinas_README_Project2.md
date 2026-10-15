# Compiling CUDA Programs
- Name: Jose Salinas
- Course: CS-6370
- Professor: Meilin Liu
- Date: 10/14/2026
- Project Name: Project 2
## Tile Matrix Multiply Program
- To compile```nvcc -DTILE_WIDTH=<Tile Width> Salinas_tileMatrixMult_Project_2.cu -o tileMatrixMult```
- To start program ```./tileMatrixMult```
- This will prompt a 1 input from the user
```
Enter Matrix Size: <Matrix Width>
```
- After it will output full matrix if matrix size  <= 8, otherwise it will output the first 20 elements. It will then verify that CPU and GPU outputs are the same.



