/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import java.util.ArrayList;
import java.util.List;
import soot.rbclassload.EntryPointDetector;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.HierarchySootMethod;

public class ThreadRunEntryPointDetector implements EntryPointDetector {

  private List<String> m_entryPoints;
  
  public ThreadRunEntryPointDetector(){
    m_entryPoints = new ArrayList<String>();
  }
  
  public void testEntryPoint(HierarchySootMethod method) {
    HierarchySootClass hclass = method.getHierarchySootClass();
    boolean found = false;
    for(String iface : hclass.getInterfaces()){
      if(iface.equals("java.lang.Runnable")){
        found = true;
      }
    }
    if(!found){
      return;
    }
    if(method.getSubSignature().equals("void run()")){
      System.out.println("found: "+method.getSignature());
      m_entryPoints.add(method.getSignature());
    }
  }

  public List<String> getEntryPoints() {
    return m_entryPoints;
  }
  
}
