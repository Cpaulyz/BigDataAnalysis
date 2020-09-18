package utils;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.MalformedURLException;
import java.util.ArrayList;

public class PDFParser {
    public static void main(String[] args) {
        try {
            // 读取PDF文件夹，将PDF格式文件路径存入一个Array中
            File dir = new File("src\\main\\resources\\ACL2020");
            ArrayList<String> targets = new ArrayList<String>();
            for(File file:dir.listFiles()){
                if(file.getAbsolutePath().endsWith(".pdf")){
                    targets.add(file.getAbsolutePath());
                }
            }
            // readPdf为提取方法
            for(String path:targets){
                readPdf(path);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 传入一个.pdf文件
     * @param file
     * @throws Exception
     */
    public static void readPdf(String file) throws Exception {
        // 是否排序
        boolean sort = false;
        // pdf文件名
        File pdfFile = new File(file);
        // 输入文本文件名称
        String textFile = null;
        // 编码方式
        String encoding = "UTF-8";
        // 开始提取页数
        int startPage = 1;
        // 结束提取页数
        int endPage = Integer.MAX_VALUE;
        // 文件输入流，生成文本文件
        Writer output = null;
        // 内存中存储的PDF Document
        PDDocument document = null;
        try {
            try {
                //注意参数已不是以前版本中的URL.而是File。
                document = PDDocument.load(pdfFile);
                // 以原来PDF的名称来命名新产生的txt文件
                if (file.length() > 4) {
                    File outputFile = new File(file.substring(0, file.length() - 4)+ ".txt");
                    textFile ="src\\main\\resources\\ACL2020\\"+outputFile.getName();
                }
            } catch (MalformedURLException e) {
                // 如果作为URL装载得到异常则从文件系统装载
                //注意参数已不是以前版本中的URL.而是File。
                document = PDDocument.load(pdfFile);
                if (pdfFile.length() > 4) {
                    textFile = file.substring(0, file.length() - 4)+ ".txt";
                }
            }
            // 文件输入流，写入文件倒textFile
            output = new OutputStreamWriter(new FileOutputStream(textFile),encoding);
            // PDFTextStripper来提取文本
            PDFTextStripper stripper = null;
            stripper = new PDFTextStripper();
            // 设置是否排序
            stripper.setSortByPosition(sort);
            // 设置起始页
            stripper.setStartPage(startPage);
            // 设置结束页
            stripper.setEndPage(endPage);
//             调用PDFTextStripper的writeText提取并输出文本
            stripper.writeText(document, output);

            System.out.println(textFile + " 输出成功！");
        } finally {
            if (output != null) {
                // 关闭输出流
                output.close();
            }
            if (document != null) {
                // 关闭PDF Document
                document.close();
            }
        }
    }
}
