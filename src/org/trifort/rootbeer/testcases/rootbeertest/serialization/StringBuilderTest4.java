/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.test.TestSerialization;

public class StringBuilderTest4 implements TestSerialization {
  public static final int N = 69878;
  public static final int M = 10677;

  public List<Kernel> create() {
    // Prepare arrays
    Random rand = new Random();
    System.out.println("big_array: double[" + N + "][" + M + "]");
    double[][] big_array = new double[N][M];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        int choice = rand.nextInt(2);
        if (choice == 0) { // 50% chance
          big_array[i][j] = rand.nextDouble();
        }
      }
    }
    System.out.println("array2: double[" + N + "]");
    double[] array2 = new double[N];
    for (int i = 0; i < N; i++) {
      array2[i] = rand.nextDouble();
    }
    System.out.println("array2: double[" + M + "]");
    double[] array3 = new double[M];
    for (int i = 0; i < M; i++) {
      array3[i] = rand.nextDouble();
    }

    List<Kernel> ret = new ArrayList<Kernel>();
    ret.add(new StringBuilderRunOnGpu4(big_array, array2, array3));
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    StringBuilderRunOnGpu4 lhs = (StringBuilderRunOnGpu4) original;
    StringBuilderRunOnGpu4 rhs = (StringBuilderRunOnGpu4) from_heap;
    return lhs.compare(rhs);
  }

}
