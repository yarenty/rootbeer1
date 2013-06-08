/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.deadmethods2;

import java.util.ArrayList;
import java.util.List;

public class MethodAnnotator {
  
  public void parse(List<Block> blocks, List<String> method_names){
    for(Block block : blocks){
      if(block.isMethod() == false){
        continue;
      }
      parseMethod(block, method_names);
    }
  }

  private void parseMethod(Block block, List<String> method_names) {
    String str = block.getFullStringNoStrings();
    String block_name = block.getMethod().getName();
    
    List<String> invoked = new ArrayList<String>();
    for(String method_name : method_names){
      if(method_name.equals(block_name)){
        continue;
      }
      int pos = str.indexOf(method_name);
      if(pos == -1 || pos == 0){
        continue;
      }
      if(str.charAt(pos-1) != ' '){
        continue;
      }
      pos += method_name.length();
      while(pos < str.length()){
        char c = str.charAt(pos);
        if(c == ' '){
          continue;
        } else if(c == '('){
          invoked.add(method_name);
          break;
        } else {
          break;
        }
      }
    }
    block.getMethod().setInvoked(invoked);
  }
}
