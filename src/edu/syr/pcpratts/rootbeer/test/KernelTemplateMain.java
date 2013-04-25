/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.test;

import edu.syr.pcpratts.rootbeer.testcases.rootbeertest.kerneltemplate.FastMatrixTest;
import edu.syr.pcpratts.rootbeer.testcases.rootbeertest.kerneltemplate.LargeThreadIndexTest;
import java.util.ArrayList;
import java.util.List;

public class KernelTemplateMain implements TestKernelTemplateFactory {

  public List<TestKernelTemplate> getProviders() {
    List<TestKernelTemplate> ret = new ArrayList<TestKernelTemplate>();
    //ret.add(new FastMatrixTest());
    ret.add(new LargeThreadIndexTest());
    return ret;
  }

}
