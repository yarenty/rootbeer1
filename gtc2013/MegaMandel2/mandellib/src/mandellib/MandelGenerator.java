/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mandellib;

import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;

/**
 *
 * @author thorsten
 */
public class MandelGenerator {

    private static Rootbeer rb = new Rootbeer();

    public static void generate(int w, int h, double minx, double maxx, double miny, double maxy, int maxdepth, int[] pixels) {
        rb.setThreadConfig(w,h);
        MyKernel myKernel = new MyKernel(pixels, maxdepth, w, h, maxx, minx, maxy, miny);
        rb.runAll(myKernel);

        System.err.println(rb.getStats().size());
    }
}
