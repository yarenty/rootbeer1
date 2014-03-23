/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;

public class StringBuilderRunOnGpu4 implements Kernel {

  private double[][] m_big_array1;
  private double[] m_array2;
  private double[] m_array3;

  public StringBuilderRunOnGpu4(double[][] big_array1, double[] array2,
      double[] array3) {
    this.m_big_array1 = big_array1;
    this.m_array2 = array2;
    this.m_array3 = array3;
  }

  @Override
  public void gpuMethod() {
    if (RootbeerGpu.getThreadId() == 0) {
      System.out.println("I will NOT throw java.lang.RuntimeException");

      int x = 0;
      // TODO Error occurs here
      StringBuilder sb = new StringBuilder("BUT I will throw it. ");
      // sb.append(x);
      // System.out.println(sb.toString());

      // System.out.println("BUT I will throw it. " + x);
    }

    // Some dummy calculations on arrays
    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();
    double value = m_big_array1[block_idxx][thread_idxx];
    if (value != 0) {
      m_array2[thread_idxx] += m_array3[thread_idxx] * value;
    }
  }

  public boolean compare(StringBuilderRunOnGpu4 rhs) {
    return true;
  }
}
