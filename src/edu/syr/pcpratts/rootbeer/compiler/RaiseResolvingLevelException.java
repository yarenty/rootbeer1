/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler;

public class RaiseResolvingLevelException extends RuntimeException {

  private String m_ClassName;
  
  public RaiseResolvingLevelException(String name) {
    m_ClassName = name;
  }
  
  public String getClassName(){
    return m_ClassName;
  }
  
}
