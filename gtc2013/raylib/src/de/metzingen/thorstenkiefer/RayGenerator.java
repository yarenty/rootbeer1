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
            double radius, double[] vx, double[] vy,int numDimensions) {
        List<MyKernel> myKernels = new ArrayList<>(w * h);

        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                myKernels.add(new MyKernel(pixels, minx, maxx, miny, maxy,
                        w, h, spheres, light, observer, vx, vy, x, y, radius,
                        numDimensions));
            }
        }
        List<Kernel> kernels = new ArrayList<>(w * h);
        kernels.addAll(myKernels);
        rb.runAll(kernels);

        //System.err.println(rb.getStats().size());
    }
}
