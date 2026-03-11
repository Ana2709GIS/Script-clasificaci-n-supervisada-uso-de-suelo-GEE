//==================================================================
//0. Visualización de ROI (Region of Interest) 
//==================================================================
var estilo = {
  color: 'red',
  fillColor: '00000000',
  width: 2
};
Map.addLayer(roi.style(estilo), {}, 'ROI delimitado');

//Centrar mapa automáticamente en el ROI
Map.centerObject(roi, 12);  

//==================================================================
//1. Seleccion de imagenes y dataset
//==================================================================

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

var imagery = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
  .filterDate("2024-01-01", "2024-04-30")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
  .filterBounds(roi)
  .map(maskS2clouds)
  .map(function (img) {
    return img.clip(roi);
  })
  .median();

function selectBands(img){
  return img.select(['B2','B3','B4','B8','B11','B12']);
}

imagery = selectBands(imagery);

Map.addLayer(imagery, {bands:['B4','B3','B2'], min:0, max:0.3}, 'RGB');

//==================================================================
// 2. Cálculo de índices
//==================================================================

function addIndices(image){

  var ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI');
  var ndbi = image.normalizedDifference(['B11','B8']).rename('NDBI');
  var mndwi = image.normalizedDifference(['B3','B11']).rename('MNDWI');
  var ndsli = image.normalizedDifference(['B4','B11']).rename('NDSLI');

  return image.addBands([ndvi,ndbi,mndwi,ndsli]);
}

// --- Agregar índices ---
var imagery = addIndices(imagery);

//Visualización del índice de agua (MNDWI)
// Parámetros de visualización
var visMNDWI = {
  min: -1,
  max: 1,
  palette: ['red', 'yellow', 'cyan', 'blue']  // suelo - transición - agua
};

// Agregar al mapa
Map.addLayer(imagery.select('MNDWI'), visMNDWI, 'MNDWI - Índice de Agua', false);

//==================================================================
// 3. Visualización RGB simple (corregido)
//==================================================================
var rgbVis = {
  min: 0.0,
  max: 0.3,
  bands: ['B4', 'B3', 'B2']
};

//======================================================================
//4.Geometrías de entrenamiento
//======================================================================
// 1: Bosque 2:Infraestrcutura, 3: Agro, 4: Agua

//Modelo de entrenamiento Machine Learning
//Fusion de muestras
var sample = Bosque
  .merge(Infraestructura)
  .merge(Agropecuario)
  .merge(Agua)
  
  .randomColumn(); // Agrega columna aleatoria para dividir entrenamiento/prueba

// División de entrenamiento y prueba
var train = sample.filter(ee.Filter.lte("random", 0.8)); // 80% para entrenamiento
var test = sample.filter(ee.Filter.gt("random", 0.8)); // 20% para validación


// Extraer valores de pixeles
var trainSample = imagery.sampleRegions({
  collection: train,             // Puntos de entrenamiento
  scale: 10,                      // Resolución espacial (Sentinel-2 = 10 m)
  properties: ["class"],
});
var testSample = imagery.sampleRegions({
  collection: test,                    // Puntos de validación
  scale: 10,
  properties: ["class"],
});

// Diccionario de leyenda
var legend = {
  LULC_class_values: [1, 2, 3, 4],   // Valores de las clases
  // 'LULC_class_palette': ['C2B280','ae8f60', '2389da', '416422', "819A20"],
 
LULC_class_palette: [
  "0B6623",  // 1 bosque
  "FF0000",  // 2 infraestructura
  "FFD700",  // 3 agro
  "00BFFF"   // 4 agua
]
};

// Entrenamiento del modelo Random Forest
var bands = imagery.bandNames();

var rf_model = ee.Classifier.smileRandomForest({
  numberOfTrees: 100
}).train({
  features: trainSample,
  classProperty: "class",
  inputProperties: bands
});

//--Clasificar la imagen---
var classified = imagery.classify(rf_model);
Map.addLayer(classified, {min:1, max:4, palette: legend.LULC_class_palette}, 'Clasificado');

//---Evaluar precisión---
var validated = testSample.classify(rf_model);

var testAccuracy = validated.errorMatrix("class", "classification");
print("Matriz de confusión", testAccuracy);
print("Exactitud global", testAccuracy.accuracy());
print("Kappa", testAccuracy.kappa());

var lulc = imagery.classify(rf_model, "LULC").toByte().set(legend);  // Clasifica toda el área, convierte a enteros, asigna metadatos de leyenda
Map.addLayer(lulc, {}, "lulc", false);

//==================================================================
// 5. Export raster (OPCIONAL) (para la descarga, eliminar "//" del script)
//==================================================================
//Export.image.toDrive({
//  image: lulcClean,
//  description: 'LULC_RF_2024',
//  folder: 'GEE_exports',
//  fileNamePrefix: 'LULC_ValleEstre',
//  region: roi,
//  scale: 10,
//  maxPixels: 1e13
//});

//==================================================================
// 6. Raster → Vector 
//==================================================================

// agregar banda dummy para cumplir requerimiento del reducer
var lulcForVector = lulc.addBands(ee.Image.constant(1).rename('dummy'));

// vectorizar
// Raster → Vector
var lulcVector = lulcForVector.reduceToVectors({
  geometry: roi,
  scale: 10,
  geometryType: 'polygon',
  eightConnected: false,
  labelProperty: 'class',     // atributo con la clase
  reducer: ee.Reducer.first(),
  maxPixels: 1e13
});

Map.addLayer(lulcVector, {}, 'LULC vector');

// exportar
Export.table.toDrive({
  collection: lulcVector,
  description: 'LULC_RF_vector',
  folder: 'Nombre',
  fileFormat: 'SHP'
});

// ===============================
// 7. Leyenda personalizada (a la izquierda)
// ===============================
var legendPanel = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px',
    backgroundColor: 'rgba(255, 255, 255, 0.8)'
  }
});

legendPanel.add(ui.Label({
  value: 'Leyenda LULC',
  style: {
    fontWeight: 'bold',
    fontSize: '14px',
    margin: '0 0 6px 0',
    color: 'black'
  }
}));

//Paleta de colores
var legendItems = [
  {color: '0B6623', name: '1 - Cobertura boscosa'},
  {color: 'FF0000', name: '2 - Infraestructura'},
  {color: 'FFD700', name: '3 - Agropecuario'},
  {color: '00BFFF', name: '4 - Agua'}
];

// Agregar cada ítem a la leyenda
legendItems.forEach(function(item) {

  var colorBox = ui.Label({
    style: {
      backgroundColor: '#' + item.color,
      padding: '8px',
      margin: '2px',
      border: '1px solid black'
    }
  });

  var description = ui.Label({
    value: item.name,
    style: {margin: '4px 0 4px 6px'}
  });

  var legendItem = ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });

  legendPanel.add(legendItem);
});

Map.add(legendPanel);

