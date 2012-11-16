/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired;

import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;

public class SimpleSynchronizedObject {

  private int m_value;
  private int m_hello;
  
  public SimpleSynchronizedObject(){
    m_hello = 0;
    m_value = 0;
  }
  
  public synchronized void inc(){
    m_hello = 0;
    m_value++;
  }
  
  public int get(){
    return m_value;
  }

}
