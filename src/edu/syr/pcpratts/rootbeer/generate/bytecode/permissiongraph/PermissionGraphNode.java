/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.bytecode.permissiongraph;

import java.util.ArrayList;
import java.util.List;
import soot.SootClass;

public class PermissionGraphNode {

  private SootClass m_Class;
  private List<SootClass> m_Children;
  
  public PermissionGraphNode(SootClass soot_class){
    m_Class = soot_class;
    m_Children = new ArrayList<SootClass>();
  }

  public void addChild(SootClass soot_class) {
    m_Children.add(soot_class);
  }
  
  public List<SootClass> getChildren(){
    return m_Children;
  }
  
  public SootClass getSootClass(){
    return m_Class;
  }
}
