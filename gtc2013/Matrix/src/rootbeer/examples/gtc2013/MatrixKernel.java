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
    int b_columns = m_blockSize * m_gridSize;
    int a_columns = m_blockSize;
    int i = thread_idxx;
    int j = block_idxx;

    int[] a = m_a;
    int[] b = m_b;

    for(int block_i = 0; block_i < 2; ++block_i){
      for(int block_j = 0; block_j < 2; ++block_j){
        int copy_i = i % 64;
        int copy_j = i % 64 * 14;
        int a_src_i = (block_i * 64) + copy_i;
        int a_src_j = (block_j * 64) + copy_j;
        float a_value = a[a_src_i*a_columns+a_src_j];

        int b_src_i = (block_i * 64) + copy_i;
        int b_src_j = (block_j * 64 * 14) + copy_j;
        float b_value = b[b_src_i*b_columns+b_src_j];

        RootbeerGpu.setSharedFloat(copy_i * 64 + copy_j, a_value);
        RootbeerGpu.setSharedFloat(64 * 64 + copy_i * 64 + copy_j, b_value);

        int i_mod = i % 64;
        int j_mod = j % 64 * 14;
        int sum = 0;
        for(int k = 0; k < 64; ++k){
          a_value = RootbeerGpu.getSharedFloat(copy_i * 64 + k);
          b_value = RootbeerGpu.getSharedFloat(copy_j * 64 + k);
          sum += a_value * b_value;
        }
        m_c[i*b_columns+j] += sum;
      }
    }
  }
}
