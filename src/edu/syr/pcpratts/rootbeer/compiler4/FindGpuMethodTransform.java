/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler4;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import soot.*;
import soot.util.Chain;

public class FindGpuMethodTransform extends BodyTransformer {

  private List<String> m_classes;
  
  public FindGpuMethodTransform(){
    m_classes = new ArrayList<String>();
  }
  
  @Override
  protected void internalTransform(Body body, String string, Map map) {
    SootMethod method = body.getMethod();
    SootClass soot_class = method.getDeclaringClass();
    SootClass kernel_class = Scene.v().getSootClass("edu.syr.pcpratts.rootbeer.runtime.Kernel");
    Chain<SootClass> ifaces = soot_class.getInterfaces();
    if(ifaces.contains(kernel_class)){
      m_classes.add(soot_class.getName());
    }
  }
  
  public List<String> getKernelClasses(){
    return m_classes;
  }
}
