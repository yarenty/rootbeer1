package org.trifort.rootbeer.entry;

import java.util.Map;
import java.util.TreeMap;

public class CubinFiles {

  private static CubinFiles instance;
  
  public static CubinFiles v(){
    if(instance == null){
      instance = new CubinFiles();
    }
    return instance;
  }
  
  private Map<String, byte[]> cubinFiles;
  
  private CubinFiles(){
    cubinFiles = new TreeMap<String, byte[]>();
  }
  
  public void put(String filename, byte[] contents){
    cubinFiles.put(filename, contents);
  }
  
  public Map<String, byte[]> getCubinFiles(){
    return cubinFiles;
  }
}
