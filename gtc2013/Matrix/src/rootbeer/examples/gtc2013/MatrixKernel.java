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

    int block_size = m_blockSize;
    int grid_size = m_gridSize;
    int block_iters = m_blockIters;

    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();

    int thread_row = thread_idxx / 32;
    int thread_col = thread_idxx % 32;

    int[] a = m_a;
    int[] b = m_b;
    int[] c = m_c;

    int sub_matrix_size = block_size / 32;
    sub_matrix_size *= sub_matrix_size;

    int m_size = block_size / 32;

    for(int block_iter = 0; block_iter < block_iters; ++block_iter){ 
      for(int sub_matrix = 0; sub_matrix < sub_matrix_size; ++sub_matrix){
        int sum = 0;
        int sub_matrix_row = sub_matrix / 2;
        int sub_matrix_col = sub_matrix % 2;
        for(int m = 0; m < m_size; ++m){
          int a_src_row = (sub_matrix_row * 32) + thread_row;
          int a_src_col = (m * 32) + thread_col;
          int a_src = (a_src_row * block_size) + a_src_col;

          int b_src = a_src;
            
          int a_value = a[a_src];
          int b_value = b[b_src];
          
          RootbeerGpu.setSharedInteger(thread_idxx * 4, a_value);
          RootbeerGpu.setSharedInteger((1024 + thread_idxx) * 4, b_value);
          RootbeerGpu.synchthreads();

          for(int k = 0; k < 32; ++k){
            a_value = RootbeerGpu.getSharedInteger((thread_row * 32 + k) * 4);
            b_value = RootbeerGpu.getSharedInteger((1024 + thread_col * 32 + k) + 4);    
            b_value = 2;     
            sum += a_value * b_value;
          }

          RootbeerGpu.synchthreads();
        }
        int dest_index = (block_iter * block_size * block_size * grid_size) + (block_idxx * block_size * block_size) + (1024 * sub_matrix) + thread_idxx;
        c[dest_index] += sum;
      }
    }
  }
}
