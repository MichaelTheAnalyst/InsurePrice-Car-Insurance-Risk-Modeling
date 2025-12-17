output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "acr_admin_username" {
  value     = azurerm_container_registry.acr.admin_username
  sensitive = true
}

output "acr_admin_password" {
  value     = azurerm_container_registry.acr.admin_password
  sensitive = true
}

output "aci_name" {
  value = azurerm_container_group.aci.name
}

output "aci_fqdn" {
  value = "http://${azurerm_container_group.aci.fqdn}:8501"
}

output "aci_ip_address" {
  value = azurerm_container_group.aci.ip_address
}
