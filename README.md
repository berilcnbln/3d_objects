# 3d_objects
excel_table.py, icp_calculations.py dosyasını import ederek excel dosyasını oluşturuyor. 



file_reader.py igs dosyalarını ply dosyalarına dönüştürüyor. file_open'ı import ederek buradan igs dosyalarını alıyor.



boundry_classification.py clustering algoritmasını kullanarak oluşan şekilden siyah noktaların olduğu pointleri çekiyor. 
rest burada cluster edildikten sonra, point cloud olarak siyah noktaları temsil ediyor.
pcl point cloudun tamamını temsil ediyor.
pcd_tree.search_radius_vector_3d(pointt, distances) metodu ile rest içindeki noktaların komşularını hesaplanıyor.
komşularının hangi segmentte kaç tane olduğunu tutan arraylleri sonunda karşılaştırarak komşuları benzer veya aynı olan noktaları bir segmentte topluyor.
daha sonra onları aynı renge boyuyor.
