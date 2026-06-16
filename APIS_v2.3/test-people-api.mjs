import { google } from 'googleapis';
import dotenv from 'dotenv';

// Cargar variables de entorno
dotenv.config();

/**
 * Script para probar la Google People API individualmente
 * Verifica si un contacto específico está agendado usando las credenciales proporcionadas
 */

// Credenciales del usuario (del registro de google_auth_temp)
const CREDENTIALS = {
    accessToken: 'ya29.a0AS3H6NzRpgZFHmC3vNIwdoBqDeoZO4v0mCNx9hRAEHzOxEzaPJTK8Sl73SoD31Yy3T01sm35gmTdhwCdSiO7Hx2l12pgijV8tTtjYZoeSyXQR0ufD0jxRB64RbDiXhPTmpi6YoCjpAugyJ9-0IUZNpdCam8mM0u3rTFdzFZBccRCCR_SgRpNquG8C4LRRTmxz9j38W0aCgYKAVYSARASFQHGX2Mic5XEZDkGhq-uQ_FqcrTkyA0206',
    refreshToken: '1//090AR9_gXpi6TCgYIARAAGAkSNwF-L9Ir7ERI_6bSt7P09Vm8232V65skkKGCXTvOe72lZXCM37FAlZs7XoMzZdOqOnvvb1MvBls',
    tokenExpiry: 1756955983527,
    authDate: new Date('2025-09-04T02:19:44.528Z')
};

// Número de teléfono a verificar (cambia este número por uno que esté en tu agenda)
const PHONE_TO_CHECK = '+57 300 2167901'; // Número corregido según la captura de pantalla

/**
 * Configurar cliente OAuth2
 */
function setupOAuth2Client() {
    const oauth2Client = new google.auth.OAuth2(
        process.env.GOOGLE_CLIENT_ID,
        process.env.GOOGLE_CLIENT_SECRET,
        process.env.GOOGLE_REDIRECT_URI
    );

    // Configurar credenciales
    oauth2Client.setCredentials({
        access_token: CREDENTIALS.accessToken,
        refresh_token: CREDENTIALS.refreshToken,
        expiry_date: CREDENTIALS.tokenExpiry
    });

    return oauth2Client;
}

/**
 * Normalizar número de teléfono para comparación
 */
function normalizePhoneNumber(phone) {
    if (!phone) return '';
    // Remover todos los caracteres no numéricos
    let normalized = phone.replace(/\D/g, '');
    
    // Si empieza con 57 y tiene 12 dígitos, mantenerlo
    if (normalized.startsWith('57') && normalized.length === 12) {
        return normalized;
    }
    
    // Si tiene 10 dígitos (número local), agregar código de país 57
    if (normalized.length === 10) {
        return '57' + normalized;
    }
    
    // Si empieza con 1 seguido de 57 (formato internacional), remover el 1
    if (normalized.startsWith('157') && normalized.length === 13) {
        return normalized.substring(1);
    }
    
    return normalized;
}

/**
 * Buscar contacto en Google Contacts
 */
async function searchContactInGoogle(phoneNumber) {
    try {
        console.log('🔍 Iniciando búsqueda de contacto...');
        console.log('📱 Número a buscar:', phoneNumber);
        console.log('📱 Número normalizado:', normalizePhoneNumber(phoneNumber));
        
        const oauth2Client = setupOAuth2Client();
        const people = google.people({ version: 'v1', auth: oauth2Client });
        
        console.log('🔑 Verificando credenciales...');
        console.log('🔑 Access Token:', CREDENTIALS.accessToken ? 'Presente' : 'Ausente');
        console.log('🔑 Refresh Token:', CREDENTIALS.refreshToken ? 'Presente' : 'Ausente');
        console.log('🔑 Token Expiry:', new Date(CREDENTIALS.tokenExpiry).toISOString());
        console.log('🔑 Token válido hasta:', new Date(CREDENTIALS.tokenExpiry) > new Date() ? 'SÍ' : 'NO');
        
        // Obtener conexiones (contactos)
        console.log('📞 Obteniendo contactos de Google...');
        let allConnections = [];
        let nextPageToken = null;
        let pageCount = 0;
        
        do {
            const response = await people.people.connections.list({
                resourceName: 'people/me',
                personFields: 'names,phoneNumbers,emailAddresses',
                pageSize: 1000,
                pageToken: nextPageToken
            });
            
            if (response.data.connections) {
                allConnections = allConnections.concat(response.data.connections);
            }
            
            nextPageToken = response.data.nextPageToken;
            pageCount++;
            
            console.log(`📄 Página ${pageCount}: ${response.data.connections ? response.data.connections.length : 0} contactos obtenidos`);
            
        } while (nextPageToken);
        
        const connections = allConnections;
        console.log(`📊 Total de contactos encontrados: ${connections.length}`);
        
        if (connections.length === 0) {
            console.log('⚠️ No se encontraron contactos en la cuenta');
            return {
                found: false,
                reason: 'No hay contactos en la cuenta',
                totalContacts: 0
            };
        }
        
        // Normalizar el número a buscar
        const targetPhone = normalizePhoneNumber(phoneNumber);
        console.log('🎯 Buscando número normalizado:', targetPhone);
        
        // Buscar el contacto
        let foundContact = null;
        let contactsWithPhones = 0;
        
        for (const contact of connections) {
            if (contact.phoneNumbers && contact.phoneNumbers.length > 0) {
                contactsWithPhones++;
                
                for (const phone of contact.phoneNumbers) {
                    const contactPhone = normalizePhoneNumber(phone.value);
                    
                    // Debug: mostrar comparaciones para encontrar el formato exacto
                    // if (contactsWithPhones <= 10) {
                    //     console.log(`🔍 Debug contacto ${contactsWithPhones}:`);
                    //     console.log(`   Original: ${phone.value}`);
                    //     console.log(`   Normalizado: ${contactPhone}`);
                    //     console.log(`   Target: ${targetPhone}`);
                    //     console.log(`   ¿Coincide?: ${contactPhone === targetPhone}`);
                    //     
                    //     // Buscar específicamente números que contengan 2167901
                    //     if (phone.value.includes('2167901') || contactPhone.includes('2167901')) {
                    //         console.log('🎯 ¡NÚMERO ENCONTRADO CON 2167901!');
                    //         console.log(`   Formato original: ${phone.value}`);
                    //         console.log(`   Formato normalizado: ${contactPhone}`);
                    //     }
                    // }
                    
                    if (contactPhone === targetPhone) {
                        foundContact = {
                            name: contact.names?.[0]?.displayName || 'Sin nombre',
                            phones: contact.phoneNumbers.map(p => p.value),
                            emails: contact.emailAddresses?.map(e => e.value) || []
                        };
                        console.log('\n🎉 ¡CONTACTO ENCONTRADO! 🎉');
                        console.log('🔍 Comparación exitosa:');
                        console.log('   Número buscado:', phoneNumber);
                        console.log('   Número encontrado:', phone.value);
                        console.log('   Normalizado buscado:', targetPhone);
                        console.log('   Normalizado encontrado:', contactPhone);
                        break;
                    }
                }
                
                if (foundContact) break;
            }
        }
        
        console.log(`📊 Contactos con números de teléfono: ${contactsWithPhones}`);
        
        if (foundContact) {
            console.log('✅ ¡CONTACTO ENCONTRADO EN LA AGENDA!');
            console.log('🎉 ==========================================');
            console.log('👤 NOMBRE DEL CONTACTO:', foundContact.name);
            console.log('📱 TELÉFONOS:', foundContact.phones.join(', '));
            console.log('📧 EMAILS:', foundContact.emails.length > 0 ? foundContact.emails.join(', ') : 'Sin emails');
            console.log('🎉 ==========================================');
            
            return {
                found: true,
                contact: foundContact,
                totalContacts: connections.length,
                contactsWithPhones
            };
        } else {
            console.log('❌ Contacto no encontrado');
            return {
                found: false,
                reason: 'Contacto no está en la agenda',
                totalContacts: connections.length,
                contactsWithPhones
            };
        }
        
    } catch (error) {
        console.error('❌ Error buscando contacto:', error);
        
        if (error.code === 401) {
            console.error('🔑 Error de autenticación - Token inválido o expirado');
        } else if (error.code === 403) {
            console.error('🚫 Error de permisos - Acceso denegado a la API');
        } else if (error.code === 429) {
            console.error('⏰ Límite de rate excedido - Demasiadas solicitudes');
        }
        
        return {
            found: false,
            error: error.message,
            code: error.code
        };
    }
}

/**
 * Función principal
 */
async function main() {
    console.log('🚀 === PRUEBA DE GOOGLE PEOPLE API ===');
    console.log('📅 Fecha de prueba:', new Date().toISOString());
    console.log('🔧 Variables de entorno:');
    console.log('   - GOOGLE_CLIENT_ID:', process.env.GOOGLE_CLIENT_ID ? 'Configurado' : 'NO CONFIGURADO');
    console.log('   - GOOGLE_CLIENT_SECRET:', process.env.GOOGLE_CLIENT_SECRET ? 'Configurado' : 'NO CONFIGURADO');
    console.log('   - GOOGLE_REDIRECT_URI:', process.env.GOOGLE_REDIRECT_URI || 'NO CONFIGURADO');
    console.log('');
    
    const result = await searchContactInGoogle(PHONE_TO_CHECK);
    
    console.log('');
    console.log('📋 === RESULTADO FINAL ===');
    console.log('🎯 Número buscado:', PHONE_TO_CHECK);
    console.log('✅ Contacto encontrado:', result.found ? 'SÍ' : 'NO');
    
    if (result.found) {
        console.log('🎊 ¡ÉXITO! El contacto SÍ está en tu agenda de Google');
        console.log('👤 NOMBRE COMPLETO:', result.contact.name);
        console.log('📱 NÚMEROS REGISTRADOS:', result.contact.phones.join(', '));
        console.log('📧 CORREOS ELECTRÓNICOS:', result.contact.emails.length > 0 ? result.contact.emails.join(', ') : 'Sin emails registrados');
    } else if (result.error) {
        console.log('❌ Error:', result.error);
        console.log('🔢 Código de error:', result.code);
    } else {
        console.log('ℹ️ Razón:', result.reason);
    }
    
    if (result.totalContacts !== undefined) {
        console.log('📊 Total de contactos en la cuenta:', result.totalContacts);
    }
    if (result.contactsWithPhones !== undefined) {
        console.log('📱 Contactos con teléfonos:', result.contactsWithPhones);
    }
    
    console.log('🏁 Prueba completada');
}

// Ejecutar la prueba
main().catch(console.error);