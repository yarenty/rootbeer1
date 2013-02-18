package rootbeer.examples.gtc2013;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;

public class MatrixKernel implements Kernel {

  private int[] m_a;
  private int[] m_b;
  private int[] m_c;
  private int m_blockSize;
  private int m_gridSize;
  private int m_blockIters;

  public MatrixKernel(int[] a, int[] b, int[] c, int block_size, int grid_size,
    int block_iters){
    m_a = a;
    m_b = b;
    m_c = c;
    m_blockSize = block_size;
    m_gridSize = grid_size;
    m_blockIters = block_iters;
  }

  public void gpuMethod(){

    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();
    int bc_columns = m_blockSize * m_gridSize * m_blockIters;
    int a_columns = m_blockSize * m_blockIters;

    int thread_row = thread_idxx / 32;
    int thread_col = thread_idxx % 32;

    int[] a = m_a;
    int[] b = m_b;
    int[] c = m_c;

    int sub_matrix_size = m_blockSize / 32;
    sub_matrix_size *= sub_matrix_size;

    int m_size = m_blockSize / 32;
    int block_iters = m_blockIters;

    for(int block_iter = 0; block_iter < block_iters; ++block_iter){ 
      for(int sub_matrix = 0; sub_matrix < sub_matrix_size; ++sub_matrix){
        int sum = 0;
        for(int m = 0; m < m_size; ++m){
          int a_src_row = thread_row;
          int a_src_col = m * 32 + thread_col;
          int a_src = a_src_row * a_columns + a_src_col;

          int b_src_row = m * 32 + thread_row;
          int b_src_col = thread_col;
          int b_src = b_src_row * bc_columns + b_src_col;
            
          float a_value = a[a_src];
          float b_value = b[b_src];

          RootbeerGpu.setSharedFloat(thread_idxx, a_value);
          RootbeerGpu.setSharedFloat(32 * 32 + thread_idxx, b_value);

          RootbeerGpu.synchthreads();

          for(int k = 0; k < 32; ++k){
            a_value = RootbeerGpu.getSharedFloat(thread_row * 32 + k);
            b_value = RootbeerGpu.getSharedFloat(32 * 32 + thread_col * 32 + k);
            a_value = 2;
            b_value = 2;
            sum += a_value * b_value;
          }

          RootbeerGpu.synchthreads();
        }
        int dest_index = (block_iter * m_blockSize * m_blockSize * m_gridSize) + (block_idxx * m_blockSize * m_blockSize) + (1024 * sub_matrix) + thread_idxx;
        c[dest_index] += sum;
      }
    }
  }
}
