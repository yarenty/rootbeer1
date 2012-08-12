/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler4;

import edu.syr.pcpratts.rootbeer.compiler.Transform2;
import java.util.List;
import java.util.Map;
import soot.SceneTransformer;

public class RootbeerTransform extends SceneTransformer {

  private List<String> m_kernelClasses;
  
  public RootbeerTransform(List<String> kernel_classes){
    m_kernelClasses = kernel_classes;
  }
  
  @Override
  protected void internalTransform(String string, Map map) {
    for(String kernel_class : m_kernelClasses){
      Transform2 transform = new Transform2();
      transform.run(kernel_class);
    }
  }
  
}
