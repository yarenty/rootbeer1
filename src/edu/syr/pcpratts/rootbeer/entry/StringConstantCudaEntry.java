/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import java.util.List;
import java.util.Set;
import soot.Body;
import soot.SootMethod;
import soot.Value;
import soot.ValueBox;
import soot.jimple.StringConstant;
import soot.rbclassload.ConditionalCudaEntry;
import soot.rbclassload.DfsInfo;
import soot.rbclassload.MethodSignatureUtil;

public class StringConstantCudaEntry extends ConditionalCudaEntry {

  public StringConstantCudaEntry(){
    super("<java.lang.String: void <init>(char[])>");
  }
  
  @Override
  public boolean condition(DfsInfo dfs_info) {
    Set<String> methods = dfs_info.getMethods();
    for(String method : methods){
      MethodSignatureUtil util = new MethodSignatureUtil();
      util.parse(method);
      SootMethod soot_method = util.getSootMethod();
      
      if(soot_method.isConcrete() == false){
        continue;
      }
      
      Body body = soot_method.retrieveActiveBody();
      List<ValueBox> boxes = body.getUseAndDefBoxes();
      for(ValueBox box : boxes){
        Value value = box.getValue();
        if(value instanceof StringConstant){
          return true;
        }
      }
    }
    return false;
  }
}
