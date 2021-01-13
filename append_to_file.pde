 import processing.serial.*;
 import java.io.FileWriter;
 import java.io.*;
 FileWriter fw;
 BufferedWriter bw;
 PrintWriter output;

   Serial myPort;  // Create object from Serial class
   String val;
   float float_val;
   float hum;
   float temp;
   int day = day();    // Values from 1 - 31
   int month = month();  // Values from 1 - 12
   int year = year();   // 2003, 2004, 2005, etc.
   String today_date;
   //return true if NaN
   boolean isNan(float val){
     return (val != val);
   }
     
     
    void setup() {
      
      String portName = Serial.list()[4];
      myPort = new Serial(this, portName, 9600);
      try {
     File file =new File("D:/MASTER/tdtr/github project/arduino_project/dataset.txt");

     FileWriter fw = new FileWriter(file, true);///true = append
     BufferedWriter bw = new BufferedWriter(fw);
     output = new PrintWriter(bw);
     if (!file.exists()) {
     file.createNewFile();
     //output.println("Day"+" "+"Humidity"+ " " + "Temperature");
    }


   }
  catch(IOException ioe) {
    System.out.println("Exception ");
    ioe.printStackTrace();
  }
    }
  
   String val_array[];
    void draw()
    {
      if (myPort.available()>0)
      {
     val= myPort.readStringUntil('\n');
     val_array =val.split(",");
     
     println(today_date);
     println("hum "+val_array[0]);
     println("temp "+val_array[1]);
  
   today_date=day+"/"+month+"/"+year;

      }
      delay(2000);
      if (val!=null ){
       float_val=float(val);
       hum=float(val_array[0]);
       temp=float(val_array[1]);

       
     if (!isNan(hum) && (hum<100) && !isNan(temp) && (temp<100) && today_date!=null){//not NaN
       output.println();
       output.print(today_date);
       output.print(" ");
       output.print(hum);
       output.print(" ");
       output.print(temp);
     }
  }
    }

  void keyPressed(){
    output.flush();
    output.close();
    exit();
  }
