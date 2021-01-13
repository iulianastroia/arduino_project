#include <Arduino_FreeRTOS.h> //free rtos library
#include "DHT.h"

#define DHTPIN 2     // pin for DHT 22 sensor(humidity+temperature)
#define DHTTYPE DHT22   // DHT 22  (AM2302)

const int LED = 3; //LED pin

int maxHum = 40; //should be 58/59
int maxTemp = 17; //should be 21/23

//assign value of LDR sensor to a temporary variable
int light_sensor=digitalRead(8);  

DHT dht(DHTPIN, DHTTYPE);
float h,t;
void setup()
//Initialize the Serial Monitor with 9600 baud rate
{
  Serial.begin(9600); 
  dht.begin();
  
  //input as light sensor
  pinMode(8,INPUT);
  pinMode(LED,OUTPUT);//declare led as output

 //priority=1
 xTaskCreate(turn_led_on, "TurnLedON", 100, NULL, 1, NULL);
 xTaskCreate(check_max, "CheckMax", 100, NULL, 1, NULL);


  delay(500);
   h = dht.readHumidity();//read humidity
   t = dht.readTemperature();//read temp
  
  // Check if any reads failed and exit early (to try again).
  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read values from DHT22 sensor!");
    return;
  }

  //turn led on and off
  if(light_sensor==HIGH)       //HIGH means,light got blocked
  digitalWrite(3,HIGH);        //if light is not present,LED on
  else
  digitalWrite(3,LOW);         //if light is present,LED off
  
  Serial.print(h);
  Serial.print(",");
  Serial.print(t);
  Serial.print(",");
  Serial.println();
 
 }


void loop()

{

}


static void check_max(void* pvParameters)
{      
  //check if humitidy and temperature are lower than set value
  //if not, light LED
  if(h <maxHum || t < maxTemp) {
  digitalWrite(LED, HIGH);  // turn the ledPin ON
 } else {
 digitalWrite(LED, LOW);  // turn the ledPin OFF
  }
   vTaskDelay(100/portTICK_PERIOD_MS);
}


static void turn_led_on(void* pvParameters)
{      
  if(light_sensor==HIGH) { //HIGH means,light got blocked
  digitalWrite(3,HIGH); //if light is not present,LED on
  }       
  else{
  digitalWrite(3,LOW);   
  }
   Serial.println(F("light ON"));
   vTaskDelay(100/portTICK_PERIOD_MS);
}
