import googlePeopleService from './baileys-api/services/google-people.service.js';
import fs from 'fs';
import path from 'path';

// Función para cargar tokens desde archivo
function loadTokens() {
    try {
        // Intentar cargar tokens de prueba primero
        const testTokenPath = path.join(process.cwd(), 'google_tokens_test.json');
        if (fs.existsSync(testTokenPath)) {
            const tokens = JSON.parse(fs.readFileSync(testTokenPath, 'utf8'));
            console.log('🔑 Usando tokens de prueba proporcionados');
            return tokens;
        }
        
        // Fallback a tokens normales
        const tokenPath = path.join(process.cwd(), 'google_tokens.json');
        if (fs.existsSync(tokenPath)) {
            const tokens = JSON.parse(fs.readFileSync(tokenPath, 'utf8'));
            return tokens;
        }
    } catch (error) {
        console.error('Error cargando tokens:', error.message);
    }
    return null;
}

// Función para probar la generación de patrones
function testPatternGeneration() {
    console.log('\n🧪 PROBANDO GENERACIÓN DE PATRONES DE BÚSQUEDA\n');
    
    const testNumbers = [
        '573002167901',
        '3002167901', 
        '+57 300 2167901',
        '+57 300-216-7901',
        '(300) 216-7901',
        '300.216.7901',
        '+1 555 123 4567',
        '5551234567'
    ];
    
    testNumbers.forEach(number => {
        console.log(`📞 Número: ${number}`);
        const patterns = googlePeopleService.generateSearchPatterns(number);
        console.log(`   Patrones generados (${patterns.length}):`);
        patterns.forEach((pattern, index) => {
            console.log(`   ${index + 1}. "${pattern}"`);
        });
        console.log('');
    });
}

// Función para probar búsqueda de contactos con optimización
async function testContactSearch() {
    console.log('\n🔍 PROBANDO BÚSQUEDA DE CONTACTOS OPTIMIZADA\n');
    
    const tokens = loadTokens();
    if (!tokens) {
        console.error('❌ No se pudieron cargar los tokens de Google');
        console.log('💡 Ejecuta primero: node test-google-auth-renewal.mjs');
        return;
    }
    
    // Números de prueba en diferentes formatos
    const testNumbers = [
        '573002167901',      // Formato webhook típico
        '3002167901',        // Solo número local
        '+57 300 2167901'    // Formato internacional con espacios
    ];
    
    for (const phoneNumber of testNumbers) {
        console.log(`\n🔍 Buscando (optimizado): ${phoneNumber}`);
        console.log('=' .repeat(50));
        
        const startTime = Date.now();
        
        try {
            const result = await googlePeopleService.searchContactByPhone(
                phoneNumber,
                tokens.access_token,
                tokens.refresh_token,
                tokens.expiry_date,
                async (newTokens) => {
                    console.log('🔄 Tokens renovados durante la búsqueda');
                    // Aquí normalmente actualizarías la base de datos
                }
            );
            
            const endTime = Date.now();
            const duration = endTime - startTime;
            
            if (result.found) {
                console.log('✅ CONTACTO ENCONTRADO:');
                console.log(`   Nombre: ${result.contact.name}`);
                console.log(`   Teléfono: ${result.contact.phoneNumber}`);
                console.log(`   Tipo: ${result.contact.formattedName}`);
                console.log(`   Patrón coincidente: "${result.matchedPattern}"`);
                console.log(`   Búsqueda ${testNumbers.indexOf(phoneNumber) + 1}: ${duration}ms - Encontrado`);
            } else {
                console.log('❌ CONTACTO NO ENCONTRADO');
                console.log(`   Búsqueda ${testNumbers.indexOf(phoneNumber) + 1}: ${duration}ms - No encontrado`);
                if (result.error) {
                    console.log(`   Error: ${result.error}`);
                }
                if (result.needsReauth) {
                    console.log('🔄 Se requiere re-autenticación');
                }
            }
        } catch (error) {
            const endTime = Date.now();
            const duration = endTime - startTime;
            console.error('❌ Error en la búsqueda:', error.message);
            console.log(`   Búsqueda ${testNumbers.indexOf(phoneNumber) + 1}: ${duration}ms - Error`);
        }
        
        console.log('');
    }
}

// Función para probar rendimiento
async function testPerformance() {
    console.log('\n⚡ PROBANDO RENDIMIENTO\n');
    
    const tokens = loadTokens();
    if (!tokens) {
        console.error('❌ No se pudieron cargar los tokens de Google');
        return;
    }
    
    const phoneNumber = '573002167901';
    const iterations = 3;
    
    console.log(`🏃‍♂️ Ejecutando ${iterations} búsquedas de "${phoneNumber}"...`);
    
    const times = [];
    
    for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();
        
        try {
            const result = await googlePeopleService.searchContactByPhone(
                phoneNumber,
                tokens.access_token,
                tokens.refresh_token,
                tokens.expiry_date
            );
            
            const endTime = Date.now();
            const duration = endTime - startTime;
            times.push(duration);
            
            console.log(`   Búsqueda ${i + 1}: ${duration}ms - ${result.found ? 'Encontrado' : 'No encontrado'}`);
        } catch (error) {
            console.error(`   Búsqueda ${i + 1}: Error - ${error.message}`);
        }
    }
    
    if (times.length > 0) {
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        
        console.log('\n📊 ESTADÍSTICAS DE RENDIMIENTO:');
        console.log(`   Tiempo promedio: ${avgTime.toFixed(2)}ms`);
        console.log(`   Tiempo mínimo: ${minTime}ms`);
        console.log(`   Tiempo máximo: ${maxTime}ms`);
    }
}

// Función principal
async function main() {
    console.log('🚀 INICIANDO PRUEBAS DEL SERVICIO MEJORADO DE GOOGLE PEOPLE API');
    console.log('=' .repeat(70));
    
    // Probar generación de patrones
    testPatternGeneration();
    
    // Probar búsqueda de contactos
    await testContactSearch();
    
    // Probar rendimiento
    await testPerformance();
    
    console.log('\n✅ PRUEBAS COMPLETADAS');
    console.log('=' .repeat(70));
}

// Ejecutar pruebas
main().catch(error => {
    console.error('❌ Error en las pruebas:', error);
    process.exit(1);
});