package rootbeer.examples.gtc2013;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;

public class MatrixKernel implements Kernel {

  private int[] m_a;
  private int[] m_b;
  private int[] m_c;
  private int m_blockSize;
  private int m_gridSize;

  public MatrixKernel(int[] a, int[] b, int[] c, int block_size, int grid_size){
    m_a = a;
    m_b = b;
    m_c = c;
    m_blockSize = block_size;
    m_gridSize = grid_size;
  }

  public void gpuMethod(){

    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();
    int bc_columns = m_blockSize * m_gridSize;
    int a_columns = m_blockSize;
    int i = thread_idxx;
    int j = block_idxx;

    int thread_row = i / 32;
    int thread_col = i % 32;

    int[] a = m_a;
    int[] b = m_b;
    int[] c = m_c;

    //System.out.println("block_idxx: "+block_idxx+" thread_idxx: "+thread_idxx);

    for(int block_i = 0; block_i < 2; ++block_i){
      for(int block_j = 0; block_j < 2; ++block_j){
        for(int m = 0; m < 2; ++m){
          int bc_col_start = 32 * 32 * block_idxx;

          int start_row = 32 * block_i;
          int start_col = 32 * block_j;                     

          int src_row = start_row + thread_row;
          int src_col_a = start_col + thread_col;   
          int src_col_bc = bc_col_start + src_col_a;

          float a_value = a[src_row * a_columns + src_col_a];
          float b_value = b[src_row * bc_columns + src_col_bc];

          //RootbeerGpu.setSharedFloat(thread_idxx, a_value);
          //RootbeerGpu.setSharedFloat(64 * 64 + thread_idxx, b_value);

          RootbeerGpu.synchthreads();

          int sum = 0;
          for(int k = 0; k < 32; ++k){
            //a_value = RootbeerGpu.getSharedFloat(thread_row * 64 + k);
            //b_value = RootbeerGpu.getSharedFloat(64 * 64 + thread_col * 64 + k);
            a_value = 2;
            b_value = 2;
            sum += a_value * b_value;
          }

          int dest_row = start_row + thread_row;
          int dest_col = bc_col_start + start_col + thread_col;
          int dest_index = dest_row * bc_columns + dest_col;
 
          c[dest_index] += sum;

          RootbeerGpu.synchthreads();
        }
      }
    }
  }
}
