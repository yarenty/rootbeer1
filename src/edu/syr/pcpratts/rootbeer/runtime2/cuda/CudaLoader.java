/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime2.cuda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class CudaLoader {

  private List<String> m_LibCudas;
  private List<String> m_Rootbeers;
  
  public CudaLoader(){
    m_LibCudas = new ArrayList<String>();
    m_LibCudas.add("/usr/lib64/libcuda.so");
    m_LibCudas.add("C:\\Windows\\System32\\nvcuda.dll");
    
    m_Rootbeers = new ArrayList<String>();
    if(File.separator.equals("/")){
      m_Rootbeers.add("cudaruntime.so.1");
      extract("cudaruntime.so.1");
    } else {
      m_Rootbeers.add("cudaruntime.dll");
      extract("cudaruntime.dll");
    }
  }
  
  public void load(){   
    doLoad(m_LibCudas);
    doLoad(m_Rootbeers);
  }

  private void doLoad(List<String> paths) {
    for(String path : paths){
      File file = new File(path);
      if(file.exists()){
        System.load(file.getAbsolutePath());
        return;
      }
    }
  }

  private void extract(String filename) {
    String path = "/edu/syr/pcpratts/rootbeer/runtime2/native/"+filename;
    try {
      InputStream is = CudaLoader.class.getResourceAsStream(path);
      if(is == null){
        path = "src"+path;
        is = new FileInputStream(path);
      }
      OutputStream os = new FileOutputStream(filename);
      while(true){
        byte[] buffer = new byte[32*1024];
        int len = is.read(buffer);
        if(len == -1)
          break;
        os.write(buffer, 0, len);
      }
      os.flush();
      os.close();
      is.close();
    } catch(Exception ex){
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }
}
