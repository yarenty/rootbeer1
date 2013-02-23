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
  private int m_blockIters;
  private Stopwatch m_cpuWatch;
  private Stopwatch m_gpuWatch;

  public MatrixApp(){
    m_cpuWatch = new Stopwatch();
    m_gpuWatch = new Stopwatch();
  }

  public void init(){
    m_blockIters = 1;
    m_blockSize = 128;
    m_gridSize = 1;
    m_a = new int[m_blockSize*m_blockSize];
    m_b = new int[m_blockSize*m_blockSize*m_gridSize*m_blockIters];
    m_ccpu = new int[m_blockSize*m_blockSize*m_gridSize*m_blockIters];
    m_cgpu = new int[m_blockSize*m_blockSize*m_gridSize*m_blockIters];

    for(int i = 0; i < m_a.length; ++i){
      m_a[i] = i %2;
    }

    for(int i = 0; i < m_b.length; ++i){
      m_b[i] = i % 2;
    }

    //printMatrix(m_a, m_blockSize);
    printRow(m_a, m_blockSize, 0);
    printCol(m_b, m_blockSize, 0);
  }

  private void printMatrix(int[] matrix, int block_size, String heading){
    System.out.println(heading);
    int row_count = 0;
    for(int i = 0; i < matrix.length; ++i){
      System.out.print(matrix[i]+" ");
      row_count++;
      if(row_count == block_size / 2){
        System.out.println();
      } else if(row_count == block_size){
        row_count = 0;
        System.out.println();
        System.out.println();
      }
    } 
  }

  private void printRow(int[] matrix, int block_size, int row){
    System.out.println("row: "+row);
    int start = row * block_size;
    for(int i = 0; i < block_size; ++i){
      System.out.print(matrix[start+i]);
    }
    System.out.println();
  }

  private void printCol(int[] matrix, int block_size, int col){
    System.out.println("col: "+col);
    for(int i = 0; i < block_size; ++i){
      System.out.print(matrix[(i * block_size) + col]);
    }
    System.out.println();
  }

  private void cpuRun(){
    int num_cores = Runtime.getRuntime().availableProcessors();
    m_cpuWatch.start();
    List<MatrixCpuThread> threads = new ArrayList<MatrixCpuThread>();
    for(int i = 0; i < num_cores; ++i){
      MatrixCpuThread thread = new MatrixCpuThread(m_a, m_b, m_ccpu, i,
        m_blockSize, m_gridSize*m_blockIters, num_cores);
      threads.add(thread);
    }
    for(int i = 0; i < num_cores; ++i){
      MatrixCpuThread thread = threads.get(i);
      thread.join();
    }
    m_cpuWatch.stop();
    System.out.println("avg cpu time: "+m_cpuWatch.getAverageTime()+" ms");
  }

  private void gpuRun(){
    m_gpuWatch.start();
    MatrixKernel matrix_kernel = new MatrixKernel(m_a, m_b, m_cgpu, m_blockSize, 
      m_gridSize, m_blockIters);
    Rootbeer rootbeer = new Rootbeer();
    rootbeer.setThreadConfig(1024, m_gridSize);
    rootbeer.runAll(matrix_kernel);
    m_gpuWatch.stop();
    System.out.println("avg gpu time: "+m_gpuWatch.getAverageTime()+" ms");

    int sum = 0;
    for(int i = 0; i < matrix_kernel.m_calcz.length; ++i){
      Calculation calc = matrix_kernel.m_calcz[i];
      if(calc != null){
        System.out.println("  calc:");
        System.out.println("    sub_matrix_row: "+calc.sub_matrix_row);
        System.out.println("    sub_matrix_col: "+calc.sub_matrix_col);
        System.out.println("    sub_matrix: "+calc.sub_matrix);
        System.out.println("    m_size: "+calc.m_size);
        System.out.println("    thread_row: "+calc.thread_row);
        System.out.println("    thread_col: "+calc.thread_col);
        System.out.println("    dest_row: "+calc.dest_row);
        System.out.println("    dest_col: "+calc.dest_col);
        System.out.println("    block_size: "+calc.block_size);
        System.out.println("    dest_index: "+calc.dest_index);
        System.out.println("    m: "+calc.m);
        System.out.println("    k: "+calc.k);
        System.out.println("    a_src_row: "+calc.a_src_row);
        System.out.println("    a_src_col: "+calc.a_src_col);
        System.out.println("    b_src_row: "+calc.b_src_row);
        System.out.println("    b_src_col: "+calc.b_src_col);
        System.out.println("    a_value: "+calc.a_value);
        System.out.println("    b_value: "+calc.b_value);
      }
    }

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
        
        //printMatrix(m_a, m_blockSize, "m_a");
        //printMatrix(m_b, m_blockSize, "m_b");
        //printMatrix(m_ccpu, m_blockSize, "m_ccpu");
        //printMatrix(m_cgpu, m_blockSize, "m_cgpu");
        System.exit(1);
        return;
      }
    }
    System.out.println("Verify PASSED!");
  }

  public void run(){
    for(int i = 0; i < 50; ++i){
      init();
      cpuRun();
      gpuRun();
      verify();
    }
  }

  public static void main(String[] args){
    Rootbeer.init();

    MatrixApp app = new MatrixApp();
    app.run();
  }
}
