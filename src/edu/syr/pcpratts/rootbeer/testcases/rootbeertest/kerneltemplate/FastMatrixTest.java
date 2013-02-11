/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.kerneltemplate;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.ThreadConfig;
import edu.syr.pcpratts.rootbeer.test.TestKernelTemplate;

public class FastMatrixTest implements TestKernelTemplate {

  private int m_a[];
  private int m_b[];
  private int m_c[];
  private int m_blockSize;
  private int m_gridSize;

  public FastMatrixTest(){
    m_blockSize = 64;
    m_gridSize = 64*14;
    m_a = new int[m_blockSize*m_blockSize];
    m_b = new int[m_blockSize*m_blockSize*m_gridSize];
    m_c = new int[m_blockSize*m_blockSize*m_gridSize];

    for(int i = 0; i < m_a.length; ++i){
      m_a[i] = i;
    }

    for(int i = 0; i < m_b.length; ++i){
      m_b[i] = i;
    }
  }

  public Kernel create() {
    Kernel ret = new MatrixKernel(m_a, m_b, m_c, m_blockSize, m_gridSize);
    return ret;
  }

  public ThreadConfig getThreadConfig() {
    ThreadConfig ret = new ThreadConfig(m_blockSize, m_gridSize);
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    MatrixKernel lhs = (MatrixKernel) original;
    MatrixKernel rhs = (MatrixKernel) from_heap;
    return lhs.compare(rhs);
  }

}
