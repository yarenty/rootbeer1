/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author thorsten
 */
public class GpuMethodTest implements TestSerialization{

    public List<Kernel> create() {
        ArrayList<Kernel> xs = new ArrayList<Kernel>();
        xs.add(new GpuMethodKernel());
        return xs;
    }

    public boolean compare(Kernel original, Kernel from_heap) {
        return true;
    }
    
}
