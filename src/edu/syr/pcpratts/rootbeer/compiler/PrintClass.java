/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler;

import edu.syr.pcpratts.rootbeer.compiler.AddBasicClass;
import edu.syr.pcpratts.rootbeer.compiler.RootbeerScene;
import edu.syr.pcpratts.rootbeer.util.JimpleWriter;
import java.util.List;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;

public class PrintClass {

  public void print(String cls){
    while(true){
      try {
        loadClass(cls);
        break;
      } catch(RuntimeException ex){
        AddBasicClass adder = new AddBasicClass();
        adder.add(ex);
      }
    }
    try {
      SootClass soot_class = Scene.v().getSootClass(cls);
      JimpleWriter writer = new JimpleWriter();
      writer.write("copied", soot_class);
    } catch(Exception ex){
      ex.printStackTrace();
    }
  }

  private void loadClass(String cls) {
    SootClass soot_class = Scene.v().getSootClass(cls);
    List<SootMethod> methods = soot_class.getMethods();
    for(SootMethod method : methods){
      if(method.isConcrete())
        method.getActiveBody();
    }
  }
  
  public static void main(String[] args){
    try {   
      String jar_filename = "../Rootbeer-Test3/dist/Rootbeer-Test3.jar";
      System.out.println("Initializing RootbeerScene");
      RootbeerScene.v().init(jar_filename);

      PrintClass printer = new PrintClass();
      printer.print("java.lang.Object");
    } catch(Exception ex){
      ex.printStackTrace();
    }
  }
}
