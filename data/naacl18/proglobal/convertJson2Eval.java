package evaluation;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;

/**
 * convert the allennlp predicted results into the required format of the evaluation script
 */
public class convertJson2Eval {
    public static void main(String[] args){
        // String systemOutput = "data/pro_global_prediction/all.chain.test.procAll.v3.pred.txt";
        // String systemEvalFile = "data/pro_global_prediction/all.chain.test.procAll.v3.pred.eval" +
        //         ".txt";
        String systemOutput = args[0].toString();
        String systemEvalFile = args[1].toString();
        evalChain1(systemOutput, systemEvalFile);
    }

    /**
     * convert the system predicted results into the format for the eval script
     * @param file : the system predicted result file
     * @param resultFile : the result file which can be directed used for eval script
     */
    public static void evalChain1(String file, String resultFile){
        try{
            BufferedReader reader = new BufferedReader(new FileReader(file));
            BufferedWriter out = new BufferedWriter(new FileWriter(resultFile));
            String line = "";
            DecimalFormat df = new DecimalFormat("0.000000");

            while((line=reader.readLine())!=null){
                JSONObject object = new JSONObject(line);

                System.out.println(line);
                String paraId = object.getString("paraid");
                String entity = object.getString("entity");
                String paragraph = object.getString("paragraph");

                String bestSpan = object.getString("best_span");
                bestSpan = bestSpan.replace("[", "");
                bestSpan = bestSpan.replace("]", "");

                while(bestSpan.contains("  "))bestSpan = bestSpan.replaceAll("  ", " ").trim();
                bestSpan = bestSpan.replaceAll("\n", "");
                String[] array = bestSpan.split(" ");
                String befStart = array[0];
                String befEnd = array[1];
                String firstAftStart = array[2];
                String firstAftEnd = array[3];

                int index = 1;

                String befLoc = "";
                String[] paraArray = paragraph.split(" ");
                if(befStart.equals("-1") && befEnd.equals("-1"))befLoc = "unk";
                else if(befStart.equals("-2") && befEnd.equals("-2"))befLoc = "null";
                else{
                    for(int i=Integer.parseInt(befStart); i<Integer.parseInt(befEnd)+1; i++){
                        befLoc += " " + paraArray[i];
                    }
                }
                befLoc = befLoc.trim();

                String aftLoc = "";
                if(firstAftStart.equals("-1") && firstAftEnd.equals("-1"))aftLoc = "unk";
                else if(firstAftStart.equals("-2") && firstAftEnd.equals("-2"))aftLoc = "null";
                else {
                    for (int i = Integer.parseInt(firstAftStart); i < Integer.parseInt(firstAftEnd) + 1; i++) {
                        aftLoc += " " + paraArray[i];
                    }
                }
                aftLoc = aftLoc.trim();

                out.write(paraId + "\t" + index + "\t" + entity + "\t" + befLoc + "\t" + aftLoc
                        + "\n");
                index++;

                String preLoc = aftLoc;
                for(int m=4; m<array.length; m++){
                    String start = array[m];
                    m += 1;
                    String end = array[m];
                    String loc = "";
                    if(start.equals("-1") && end.equals("-1"))loc = "unk";
                    else if(start.equals("-2") && end.equals("-2"))loc = "null";
                    else {
                        for (int i = Integer.parseInt(start); i < Integer.parseInt(end) + 1; i++) {
                            loc += " " + paraArray[i];
                        }
                    }
                    loc = loc.trim();
                    out.write(paraId + "\t" + index + "\t" + entity + "\t" + preLoc + "\t" + loc +
                                    "\n");
                    preLoc = loc;
                    index++;
                }
            }
            out.close();
        }catch(Exception e){
            e.printStackTrace();
        }
    }

}
