/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

/**
 *
 * @author thorsten
 */
public class GpuMethodKernel implements Kernel {

    public void gpuMethod(int a) {
        gpuMethod(1,2);
        gpuMethod(1,2,3);
    }

    public void gpuMethod(int a, int b) {
        gpuMethod(1,2,3);
    }

    public void gpuMethod(int a, int b, int c) {
    }

    public void gpuMethod() {
        gpuMethod(1);
        gpuMethod(1,2);
        gpuMethod(1,2,3);
    }
}
