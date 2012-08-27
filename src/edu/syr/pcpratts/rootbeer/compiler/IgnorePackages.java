/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler;

import java.util.ArrayList;
import java.util.List;

public class IgnorePackages {

  private List<String> m_ignorePackages;
  
  public IgnorePackages(){
    m_ignorePackages = new ArrayList<String>();
    m_ignorePackages.add("soot");
    m_ignorePackages.add("java_cup");
    m_ignorePackages.add("polyglot");
    m_ignorePackages.add("ppg");
    m_ignorePackages.add("antlr");
    m_ignorePackages.add("org.antlr");
    m_ignorePackages.add("jas");
    m_ignorePackages.add("jasmin");
    m_ignorePackages.add("scm");  
  }
  
  public List<String> get() {
    return m_ignorePackages;
  }
}
