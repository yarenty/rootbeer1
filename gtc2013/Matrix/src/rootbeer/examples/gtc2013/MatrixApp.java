package rootbeer.examples.gtc2013;

import edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.StatsRow;
import java.util.List;
import java.util.ArrayList;

//see: http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
public class MatrixApp {

  private int m_a[];
  private int m_b[];
  private int m_cgpu[];
  private int m_ccpu[];
  private int m_blockSize;
  private int m_gridSize;

  public MatrixApp(){
    m_blockSize = 256;
    m_gridSize = 256*14;
    m_a = new int[m_blockSize*m_blockSize];
    m_b = new int[m_blockSize*m_blockSize*m_gridSize];
    m_ccpu = new int[m_blockSize*m_blockSize*m_gridSize];
    m_cgpu = new int[m_blockSize*m_blockSize*m_gridSize];

    for(int i = 0; i < m_a.length; ++i){
      m_a[i] = 2;
    }

    for(int i = 0; i < m_b.length; ++i){
      m_b[i] = 2;
    }
  }

  private void cpuRun(){
    int num_cores = Runtime.getRuntime().availableProcessors();
    Stopwatch watch = new Stopwatch();
    watch.start();
    List<MatrixCpuThread> threads = new ArrayList<MatrixCpuThread>();
    for(int i = 0; i < num_cores; ++i){
      MatrixCpuThread thread = new MatrixCpuThread(m_a, m_b, m_ccpu, i,
        m_blockSize, m_gridSize, num_cores);
      threads.add(thread);
    }
    for(int i = 0; i < num_cores; ++i){
      MatrixCpuThread thread = threads.get(i);
      thread.join();
    }
    watch.stop();
    System.out.println("cpu time: "+watch.elapsedTimeMillis()+" ms");
  }

  private void gpuRun(){
    Stopwatch watch = new Stopwatch();
    watch.start();
    MatrixKernel matrix_kernel = new MatrixKernel(m_a, m_b, m_cgpu, m_blockSize, m_gridSize);
    Rootbeer rootbeer = new Rootbeer();
    rootbeer.setThreadConfig(1024, 14);
    rootbeer.runAll(matrix_kernel);
    watch.stop();
    System.out.println("gpu time: "+watch.elapsedTimeMillis()+" ms");

    List<StatsRow> stats = rootbeer.getStats();
    for(StatsRow row : stats){
      System.out.println("  StatsRow:");
      System.out.println("    init time: "+row.getInitTime());
      System.out.println("    serial time: "+row.getSerializationTime());
      System.out.println("    exec time: "+row.getExecutionTime());
      System.out.println("    deserial time: "+row.getDeserializationTime());
      System.out.println("    num blocks: "+row.getNumBlocks());
      System.out.println("    num threads: "+row.getNumThreads());
    }
  }

  private void verify(){
    for(int i = 0; i < m_ccpu.length; ++i){
      int cpu_value = m_ccpu[i];
      int gpu_value = m_cgpu[i];
      if(cpu_value != gpu_value){
        System.out.println("Verify Failed.");
        System.out.println("  cpu_value: "+cpu_value);
        System.out.println("  gpu_value: "+gpu_value);
        System.out.println("  index: "+i);
        System.exit(1);
      }
    }
    System.out.println("Verify PASSED!");
  }

  public void run(){
    cpuRun();
    gpuRun();
    verify();
  }

  public static void main(String[] args){
    Rootbeer.init();

    MatrixApp app = new MatrixApp();
    app.run();
  }
}
