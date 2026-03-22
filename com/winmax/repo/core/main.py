from warnings import catch_warnings
import os
from pyspark.sql.functions import lit
import sys
from pyspark.sql.functions import col

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

os.environ['PYSPARK_PYTHON'] = "C:\\Users\\yoursystemname\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = "C:\\Users\\yoursystemname\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
# =========================================
# 1. SPARK SESSION
# =========================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("Retail_Data_Pipeline") \
    .getOrCreate()

# =========================================
# 2. BRONZE LAYER (RAW DATA)
# =========================================

# ITEM
item_df = spark.createDataFrame([
    (1, "Laptop"), (2, "Mobile"), (3, "Tablet"),
    (4, "Headphones"), (5, "Keyboard")
], ["ITEM_ID", "ITEM_NAME"])

# VENDOR
vendor_df = spark.createDataFrame([
    (101, "TechSupplier Inc", "New York"),
    (102, "GadgetWorld", "California"),
    (103, "ElectroHub", "Texas")
], ["VENDOR_ID", "VENDOR_NAME", "VENDOR_ADDRESS"])

# PURCHASE
purchase_df = spark.createDataFrame([
    (1001, 101, "TechSupplier Inc", 1, "Laptop", "Bulk", "New York", 50, 800),
    (1002, 102, "GadgetWorld", 2, "Mobile", "Bulk", "California", 100, 300),
    (1003, 103, "ElectroHub", 3, "Tablet", "Bulk", "Texas", 70, 400),
    (1004, 101, "TechSupplier Inc", 4, "Headphones", "Retail", "New York", 200, 50),
    (1005, 102, "GadgetWorld", 5, "Keyboard", "Retail", "California", 150, 30),
    (1006, 103, "ElectroHub", 1, "Laptop", "Bulk", "Texas", 30, 780)
], ["PURCHASE_ID","VENDOR_ID","VENDOR_NAME","ITEM_ID","ITEM_NAME",
    "PURCHASE_TYPE","VENDOR_ADDRESS","QUANTITY","PRICE"])

# SALES
sales_df = spark.createDataFrame([
    (1,"Laptop",5,"2025-03-01",1000,5001,"Online"),
    (2,"Mobile",10,"2025-03-01",500,5002,"Store"),
    (3,"Tablet",7,"2025-03-02",600,5003,"Online"),
    (4,"Headphones",20,"2025-03-02",80,5004,"Store"),
    (5,"Keyboard",15,"2025-03-03",60,5005,"Online"),
    (1,"Laptop",3,"2025-03-04",950,5006,"Store"),
    (2,"Mobile",8,"2025-03-05",520,5007,"Online"),
    (3,"Tablet",4,"2025-03-06",610,5008,"Store"),
    (4,"Headphones",25,"2025-03-07",75,5009,"Online")
], ["ITEM_ID","ITEM_NAME","QUANTITY","SALE_DATE","PRICE","SALE_ORDER_ID","SALE_TYPE"])

# STOCK
stock_df = spark.createDataFrame([
    (1,"Laptop",60,"S1","W1"),
    (2,"Mobile",120,"S2","W1"),
    (3,"Tablet",80,"S3","W2"),
    (4,"Headphones",300,"S4","W2"),
    (5,"Keyboard",200,"S5","W3")
], ["ITEM_ID","ITEM_NAME","QUANTITY","STORE_SHELF_ID","WAREHOUSE_SHELF_ID"])

# BILLING
billing_df = spark.createDataFrame([
    (9001,"2025-03-01","SALE",5001,None,5000,500,5500),
    (9002,"2025-03-01","SALE",5002,None,5000,500,5500),
    (9003,"2025-03-02","SALE",5003,None,4200,420,4620),
    (9004,"2025-03-02","SALE",5004,None,1600,160,1760),
    (9005,"2025-03-03","SALE",5005,None,900,90,990),
    (9006,"2025-03-04","SALE",5006,None,2850,285,3135),
    (9101,"2025-03-01","PURCHASE",None,1001,40000,4000,44000),
    (9102,"2025-03-02","PURCHASE",None,1002,30000,3000,33000)
], ["TNS_ID","TNS_DATE","TNS_TYPE","SALE_ORDER_ID","PURCHASE_ID","AMOUNT","TAX","TOTAL_BILL_AMT"])

# =========================================
# 3. SILVER LAYER (DATA CLEANSING)
# =========================================

# Standardize
item_df = item_df.withColumn("ITEM_NAME", upper(trim(col("ITEM_NAME"))))
vendor_df = vendor_df.withColumn("VENDOR_NAME", upper(trim(col("VENDOR_NAME"))))

# Date conversion
sales_df = sales_df.withColumn("SALE_DATE", to_date("SALE_DATE"))
billing_df = billing_df.withColumn("TNS_DATE", to_date("TNS_DATE"))

# Remove duplicates
sales_df = sales_df.dropDuplicates(["ITEM_ID","SALE_ORDER_ID","SALE_DATE"])
purchase_df = purchase_df.dropDuplicates(["PURCHASE_ID"])

# Remove invalid values
sales_df = sales_df.filter((col("QUANTITY") > 0) & (col("PRICE") > 0))
purchase_df = purchase_df.filter((col("QUANTITY") > 0) & (col("PRICE") > 0))

# Normalize SALE_TYPE
sales_df = sales_df.withColumn(
    "SALE_TYPE",
    when(col("SALE_TYPE").isin("online","Online"),"ONLINE")
    .when(col("SALE_TYPE").isin("store","Store"),"STORE")
    .otherwise("OTHER")
)

# Fix vendor consistency
vendor_lookup = vendor_df.select("VENDOR_ID", col("VENDOR_NAME").alias("VN"))
purchase_df = purchase_df.join(vendor_lookup, "VENDOR_ID", "left") \
    .withColumn("VENDOR_NAME", col("VN")).drop("VN")

# Data quality flag
sales_df = sales_df.withColumn(
    "DQ_FLAG",
    when(col("PRICE").isNull(),"MISSING_PRICE")
    .when(col("QUANTITY") <= 0,"INVALID_QTY")
    .otherwise("VALID")
)

# =========================================
# 4. GOLD LAYER (REPORTING)
# =========================================

# 1. Daily Sales
daily_sales = sales_df.groupBy("SALE_DATE") \
    .agg(sum(col("QUANTITY")*col("PRICE")).alias("total_sales"))

# 2. Top Items
top_items = sales_df.groupBy("ITEM_ID") \
    .agg(sum("QUANTITY").alias("qty")).orderBy(desc("qty"))

# 3. Low Stock
low_stock = stock_df.filter(col("QUANTITY") < 50)

# 4. Purchase vs Sales
purchase_vs_sales = purchase_df.groupBy("ITEM_ID") \
    .agg(sum("QUANTITY").alias("purchased")) \
    .join(
        sales_df.groupBy("ITEM_ID")
        .agg(sum("QUANTITY").alias("sold")),
        "ITEM_ID","outer"
    ).fillna(0)

# 5. Vendor Purchase
vendor_purchase = purchase_df.groupBy("VENDOR_ID") \
    .agg(sum(col("QUANTITY")*col("PRICE")).alias("total"))

# 6. Billing Summary
billing_summary = billing_df.agg(
    sum("AMOUNT"), sum("TAX"), sum("TOTAL_BILL_AMT")
)

# 7. Running Total
window_spec = Window.partitionBy("ITEM_ID").orderBy("SALE_DATE")
running_total = sales_df.withColumn(
    "sales_amt", col("QUANTITY")*col("PRICE")
).withColumn(
    "running_total", sum("sales_amt").over(window_spec)
)

# 8. Aging Report
aging = sales_df.groupBy("ITEM_ID") \
    .agg(max("SALE_DATE").alias("last_sale")) \
    .withColumn("days_since", datediff(current_date(),"last_sale"))

# 9. Stock Reconciliation
recon = purchase_vs_sales.join(stock_df,"ITEM_ID") \
    .withColumn("expected", col("purchased")-col("sold")) \
    .withColumn("diff", col("expected")-col("QUANTITY"))

# 10. Sales-Billing Mismatch
sales_amt = sales_df.withColumn("calc_amt", col("QUANTITY")*col("PRICE"))
mismatch = sales_amt.join(billing_df,"SALE_ORDER_ID") \
    .withColumn("diff", col("calc_amt")-col("TOTAL_BILL_AMT")) \
    .filter(col("diff") != 0)

# =========================================
# 5. OUTPUT (SHOW SAMPLE)
# =========================================

daily_sales.show()
top_items.show()
low_stock.show()
purchase_vs_sales.show()
vendor_purchase.show()
billing_summary.show()
running_total.show()
aging.show()
recon.show()
mismatch.show()
