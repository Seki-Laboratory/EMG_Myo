//シリアル通信(PC⇔Arduino)
char data;

void setup() {
 Serial.begin(115200);
 pinMode(13,OUTPUT);
 digitalWrite(13,LOW);
}

void loop(){
 if (Serial.available() > 0) {
   data = Serial.read();
   if(data == '1'){
     //実行したい内容
     digitalWrite(13,HIGH);
   }
   else{
     //違う時
     digitalWrite(13,LOW);
     }
 }
}