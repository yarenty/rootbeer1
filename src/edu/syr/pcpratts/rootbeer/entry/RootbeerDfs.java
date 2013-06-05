/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import soot.SootField;
import soot.Type;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.DfsInfo;
import soot.rbclassload.FieldSignatureUtil;
import soot.rbclassload.HierarchyValueSwitch;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringToType;

/**
 *
 * @author pcpratts
 */
public class RootbeerDfs {
  
  private DfsInfo m_currDfsInfo;
  
  public void run(DfsInfo dfs_info) {
    m_currDfsInfo = dfs_info;
    String signature = dfs_info.getRootMethodSignature();
    
    Set<String> visited = new HashSet<String>();
    //System.out.println("doing rootbeer dfs: "+signature);
    LinkedList<String> queue = new LinkedList<String>();
    queue.add(signature);
    queue.add("<java.lang.Integer: java.lang.String toString(int)>");
    queue.add("<edu.syr.pcpratts.rootbeer.runtime.Sentinal: void <init>()>");
    queue.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: void <init>()>");
    queue.add("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: edu.syr.pcpratts.rootbeer.runtimegpu.GpuException arrayOutOfBounds(int,int,int)>");
    
    while(queue.isEmpty() == false){
      String curr = queue.removeFirst();
      doDfsForRootbeer(curr, queue, visited);
    }
  }

  private void doDfsForRootbeer(String signature, LinkedList<String> queue, Set<String> visited){

    if(visited.contains(signature)){
      return;
    }
    visited.add(signature);
    
    StringToType converter = new StringToType();
    FieldSignatureUtil futil = new FieldSignatureUtil();
    MethodSignatureUtil mutil = new MethodSignatureUtil();

    mutil.parse(signature);
    m_currDfsInfo.addType(mutil.getClassName());
    m_currDfsInfo.addType(mutil.getReturnType());
    m_currDfsInfo.addMethod(signature);
    
    ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
    List<String> virt_methods = class_hierarchy.getVirtualMethods(signature);
    for(String virt_method : virt_methods){
      if(RootbeerClassLoader.v().dontFollow(virt_method)){
        continue;
      }

      if(virt_method.equals(signature) == false){
    	  queue.add(virt_method);
      }
    }

    HierarchyValueSwitch value_switch = RootbeerClassLoader.v().getValueSwitch(signature);
    for(String type_str : value_switch.getAllTypes()){
      Type type = converter.convert(type_str);
      m_currDfsInfo.addType(type);
    }    

    for(String method_sig : value_switch.getMethodRefs()){
      if(RootbeerClassLoader.v().dontFollow(method_sig)){
        continue;
      }
         	      
      queue.add(method_sig);
    }

    for(String field_ref : value_switch.getFieldRefs()){
      futil.parse(field_ref);
      SootField soot_field = futil.getSootField();
      m_currDfsInfo.addField(soot_field);
    }

    for(String instanceof_str : value_switch.getInstanceOfs()){
      Type type = converter.convert(instanceof_str);
      m_currDfsInfo.addInstanceOf(type);
    }
  }
}
