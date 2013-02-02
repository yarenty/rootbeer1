package rootbeer.examples.gtc2013;

import edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch;

//see: http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
public class MatrixApp {

  private int m_a[];
  private int m_b[];
  private int m_cgpu[];
  private int m_ccpu[];
  private int m_blockSize;
  private int m_gridSize;

  public MatrixApp(){
    m_blockSize = 64;
    m_gridSize = 14;
    m_a = new int[block_size*block_size];
    m_b = new int[block_size*block_size*grid_size];
    m_ccpu = new int[block_size*block_size*grid_size];
    m_cgpu = new int[block_size*block_size*grid_size];

    for(int i = 0; i < a.length; ++i){
      m_a[i] = i;
    }

    for(int i = 0; i < b.length; ++i){
      m_b[i] = i;
    }
  }

  public int cpuRun(){
    Stopwatch watch = new Stopwatch();
    List<Thread> threads = new ArrayList<Thread>();
    for(int i = 0; i < 4; ++i){
      
    }
    watch.stop();
    System.out.println("cpu time: "+watch.elapsedTimeMillis()+" ms");
  }

  public int gpuRun(){
    Stopwatch watch = new Stopwatch();
    List<Kernel> kernels = new ArrayList<Kernal>();
    for(int i = 0; i < block_size * grid_size; ++i){
      kernels.add(new MatrixKernel(m_a, m_b, m_cgpu, m_blockSize));
    } 
    Rootbeer rootbeer = new Rootbeer();
    rootbeer.setThreadConfig(m_blockSize, m_gridSize);
    rootbeer.runAll(kernels);
    watch.stop();
    System.out.println("gpu time: "+watch.elapsedTimeMillis()+" ms");
  }

  public static void main(String[] args){
    MatrixApp app = new MatrixApp();
    app.run();
  }
}
