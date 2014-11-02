package org.trifort.rootbeer.entry;

import java.util.List;

import soot.rtaclassload.Operand;
import soot.rtaclassload.RTAClass;
import soot.rtaclassload.RTAClassLoader;
import soot.rtaclassload.RTAInstruction;
import soot.rtaclassload.RTAMethod;

public class KernelFinder {
  
  public static RTAClass find(RTAMethod method) {
    List<RTAInstruction> instructions = method.getInstructions();
    for(RTAInstruction inst : instructions){
      String name = inst.getName();
      if(name.equals("new")){
        List<Operand> operands = inst.getOperands();
        for(Operand operand : operands){
          if(operand.getType().equals("class_ref") == false){
            continue;
          }
          String class_name = operand.getValue();
          RTAClass rtaClass = RTAClassLoader.v().getRTAClass(class_name);
          List<String> ifaces = rtaClass.getInterfaceStrings();
          for(String iface : ifaces){
            if(iface.equals("org.trifort.rootbeer.runtime.Kernel")){
              return rtaClass;
            }
          }
        }
      }
    }
    return null;
  }
}
