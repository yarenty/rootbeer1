/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler;

import edu.syr.pcpratts.rootbeer.generate.bytecode.GenerateRuntimeBasicBlock;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLScene;
import java.util.List;
import soot.*;
import soot.rbclassload.DfsInfo;
import soot.rbclassload.RootbeerClassLoader;

public class Transform2 {
  
  private int m_Uuid;
  
  public Transform2(){
    m_Uuid = 1;
  }

  public void run(String cls){    
    if(cls.equals("edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired.NewOnGpuRunOnGpu")){
      System.out.println("hello");
    }
    DfsInfo dfs_info = RootbeerClassLoader.v().getDfsInfo();
    List<Type> types = dfs_info.getOrderedRefLikeTypes();
    OpenCLScene scene = new OpenCLScene();
    for(Type type : types){
      if(type instanceof RefType == false){
        continue;
      }
      RefType ref_type = (RefType) type;
      scene.addClass(ref_type.getSootClass());
    }
    
    OpenCLScene.setInstance(scene);
    
    System.out.println("running Transform2 on: "+cls);
    
    SootClass soot_class1 = Scene.v().getSootClass(cls);
    SootMethod method = soot_class1.getMethod("void gpuMethod()");
    
    //generate RuntimeBasicBlock and GcObjectVisitor
    String uuid = getUuid();
    GenerateRuntimeBasicBlock generate = new GenerateRuntimeBasicBlock(method, uuid);
    try {
      generate.makeClass();
    } catch(Exception ex){
      ex.printStackTrace();
      OpenCLScene.releaseV();
      return;
    }

    //add an interface to the class
    SootClass soot_class = method.getDeclaringClass();
    SootClass iface_class = Scene.v().getSootClass("edu.syr.pcpratts.rootbeer.runtime.CompiledKernel");
    soot_class.addInterface(iface_class);
    
    OpenCLScene.releaseV();
  }
  
  private String getUuid(){
    int uuid = m_Uuid;
    m_Uuid++;
    return Integer.toString(uuid);
  }
}
