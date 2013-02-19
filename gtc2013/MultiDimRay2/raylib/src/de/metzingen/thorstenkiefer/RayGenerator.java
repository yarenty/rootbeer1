/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.metzingen.thorstenkiefer;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author thorsten
 */
public class RayGenerator {

    private static Rootbeer rb = new Rootbeer();

    public static void generate(
            int w, int h, double minx, double maxx, double miny, double maxy,
            int[] pixels, double[][] spheres, double[] light, double[] observer,
            double radius, double[] vx, double[] vy, int numDimensions) {
        rb.setThreadConfig(w, h);
        MyKernel myKernel = new MyKernel(pixels, minx, maxx, miny, maxy,
                w, h, spheres, light, observer, vx, vy, radius,
                numDimensions);
        rb.runAll(myKernel);
        //System.err.println(rb.getStats().size());
    }
}
