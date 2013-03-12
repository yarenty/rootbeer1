/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mandellib;

import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch;

/**
 *
 * @author thorsten
 */
public class MandelGenerator {

  private static Rootbeer rb = new Rootbeer();
  private static Stopwatch m_gpuWatch = new Stopwatch();

  public static void gpuGenerate(int w, int h, double minx, double maxx, double miny, double maxy, int maxdepth, int[] pixels) {
    m_gpuWatch.start();
    rb.setThreadConfig(w,h);
    MyKernel myKernel = new MyKernel(pixels, maxdepth, w, h, maxx, minx, maxy, miny);
    rb.runAll(myKernel);
    m_gpuWatch.stop();
    System.out.println("avg gpu: "+m_gpuWatch.getAverageTime());    
  }
}
