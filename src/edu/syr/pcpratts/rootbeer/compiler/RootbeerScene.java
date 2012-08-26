/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler;

import java.util.ArrayList;
import java.util.List;
import soot.Scene;
import soot.SootClass;

public class RootbeerScene {

  private static RootbeerScene m_instance;
  
  public static RootbeerScene v(){
    if(m_instance == null){
      m_instance = new RootbeerScene();
    }
    return m_instance;
  }

  private List<String> m_addedClasses;
  private List<String> m_reachables;
  
  public RootbeerScene(){
    m_addedClasses = new ArrayList<String>();
  }
  
  public void addClass(SootClass new_class) {
    if(m_addedClasses.contains(new_class.getName()) == false){
      m_addedClasses.add(new_class.getName());
      Scene.v().addClass(new_class);
    }
  }

  public List<String> getAddedClasses() {
    return m_addedClasses;
  }

  public void setReachableMethods(List<String> reachables) {
    m_reachables = reachables;
  }

  public List<String> getReachableMethods() {
    return m_reachables;
  }
}
