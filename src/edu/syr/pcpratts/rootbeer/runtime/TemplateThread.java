/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.syr.pcpratts.rootbeer.runtime;

import static edu.syr.pcpratts.rootbeer.runtime.TemplateThread.sleeping;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author thorsten
 */
class TemplateThread extends Thread {

  public static List<TemplateThread> sleeping = Collections.synchronizedList(new ArrayList<TemplateThread>());
  public static List<TemplateThread> computing = Collections.synchronizedList(new ArrayList<TemplateThread>());
  public boolean compute = false;
  public int startid;
  public int endid;
  public int threadid;
  public int blockid;
  public Kernel kernel;

  static {
    for (int i = 0; i < 8; ++i) {
      TemplateThread t = new TemplateThread();
      t.start();
      sleeping.add(t);
    }
  }

  @Override
  public void run() {
    while (true) {
      while (!compute) {
        try {
          sleep(100000);
        } catch (InterruptedException ex) {
        }
      }
      computing.add(this);
      for (threadid = startid; threadid < endid; ++threadid) {
        kernel.gpuMethod();
      }
      compute = false;
      computing.remove(this);
      sleeping.add(this);
    }
  }
}
