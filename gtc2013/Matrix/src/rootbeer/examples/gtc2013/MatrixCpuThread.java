package rootbeer.examples.gtc2013;


public class MatrixCpuThread implements Runnable {

  private int[] m_a;
  private int[] m_b;
  private int[] m_c;
  private int m_index;
  private int m_blockSize;
  
  public MatrixCpuThread(int[] a, int[] b, int[] c, int index, int block_size){
    m_a = a;
    m_b = b;
    m_c = c;
    m_index = index;
    m_blockSize = block_size;
  }

  @Override
  public void run(){
    int num_each = m_b.length / 4;
    int start_row = m_index * num_each;
    int stop_row = (m_index + 1) * num_each;    
    if(m_index == 3){
      stop_row = m_b.length;
    }

    for(int b_row = start_row; b_row < stop_row; ++b_row){
      
    }
  }
}
